// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package cbtutil provides utilities for interacting with Cloud BigTable.
package cbtutil

import (
	"context"
	"fmt"
	"strconv"

	bigtable "cloud.google.com/go/bigtable"
	bttest "cloud.google.com/go/bigtable/bttest"
	option "google.golang.org/api/option"
	grpc "google.golang.org/grpc"
)

// NewConnection creates new fake connection to bigtable
func NewConnection() *bigtable.Table {
	srv, err := bttest.NewServer("localhost:0")
	try(err)
	ctx := context.Background()

	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
	try(err)

	proj, instance := "proj", "instance"

	adminClient, err := bigtable.NewAdminClient(ctx, proj, instance, option.WithGRPCConn(conn))
	try(err)

	err = adminClient.CreateTable(ctx, "randomArrayVals")
	try(err)

	err = adminClient.CreateColumnFamily(ctx, "randomArrayVals", "vals")
	try(err)

	client, err := bigtable.NewClient(ctx, proj, instance, option.WithGRPCConn(conn))
	try(err)
	tbl := client.Open("randomArrayVals")
	return tbl
}

//WriteVal writes val to tbl at specified row
func WriteVal(tbl *bigtable.Table, rowID string, val int) {
	ctx := context.Background()
	mut := bigtable.NewMutation()
	mut.Set("vals", "vals1", bigtable.Now(), []byte(strconv.Itoa(val)))
	err := tbl.Apply(ctx, rowID, mut)
	try(err)
}

//ReadVal reads val at specified row
func ReadVal(tbl *bigtable.Table, rowID string) int {
	ctx := context.Background()
	row, err := tbl.ReadRow(ctx, rowID)
	try(err)

	var retVal = 0
	for _, column := range row["vals"] {
		val, err := strconv.Atoi(string(column.Value))
		try(err)
		fmt.Println(column.Column + ":" + strconv.Itoa(val))
		retVal = val
	}
	return retVal
}

func try(err error) {
	if err != nil {
		panic(err)
	}
}
